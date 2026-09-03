# Literature review: surrogate, MDO and external-validation blockers

Scope: the three blockers that stop the cusped-field (HEMP-type) thruster rebuild from
producing an accepted learned surrogate, a physics-bearing optimisation result, or any
external validation, plus the reproducibility practice that carries all three. For each
blocker this document gives the state of the art with verified references, the documented
pitfalls, and concrete recommendations mapped to the experiments in `modern/experiments/`.
It changes no code and admits no claim; it is a reading of the literature against the
committed evidence at `origin/feat/sota-foundation` (`96220ffc`, 2026-09-03).

Citation style is author-year; every cited work is listed in the bibliography (section 8)
with a DOI, arXiv identifier or a publisher/conference URL that was opened during this
review. Verification method and the items dropped as unverifiable are in section 0.2.

## 0. Method and honesty statement

### 0.1 What was read in the repository (read-only)

`modern/docs/ROADMAP_AUDIT.md` (sections 2.2, 2.3, 2.9, 4, 5), the surrogate, optimisation,
active-learning and validation workstream documents, `wall_loss_geometry_surrogate_v1`
and `_v2` (README, `results/artifacts/assessment.json`), `mdo_l0_campaign_v1`
(`protocol.json`, `POSTHOC_AUDIT.md`), `mdo_l0_campaign_v2` (`protocol.json`, README),
`l0_surrogate_v9/POSTHOC_AUDIT.md`, the `l1a_field_surrogate_v10` README on
`origin/exp/l1a-field-surrogate-v10`, `paper/manuscript.tex` (V&V protocol, GATE-L1/L2/L3
text), `modern/docs/REFERENCES.md`, and the two PDFs in the repository root
(`ASRC - CFT MDO.pdf` is a status-report deck whose "Validation" slide reads "match
experimental results" and cites Ma et al. 2015; it contains no device measurements).
The 2026-09-03 `.cursor/memory` devlog and scratchpad were read for the PIC steady-state v2
plateau numbers and the surrogate-v2 diagnosis.

### 0.2 How references were verified, and what was omitted

Every DOI below was resolved through the Crossref REST API on 2026-09-03 and the returned
title, author list, container, volume/pages and year are what appear in the bibliography
(they were generated from the Crossref records, not typed). arXiv identifiers were resolved
through the arXiv API. Conference papers without a DOI (IEPC, ISTS, PMLR, JMLR, standards)
were opened at the publisher or conference archive URL given. Items whose full text was
read (not only metadata) are marked "(full text read)" in the bibliography.

Dropped as unverifiable or wrong, so **not** cited:

- "Ma 2024, Aerospace Science and Technology" for cusped-field-thruster MDO. The DOI
  supplied with the task (`10.1016/j.ast.2024.109516`) resolves to Yeo, Gadisa, Ogawa and
  Bang (2024), *Multi-objective design optimization and physics-based sensitivity analysis
  of field emission electric propulsion for CubeSat platforms*, AST 154:109516 — a FEEP
  paper from the same group. It is cited under that correct title as the downstream MDO
  methodology paper; no 2024 AST paper by "Ma" on cusped-field thrusters was found.
- A "TU Berlin low-power HEMP" dataset. Two searches found no TU Berlin HEMP publication;
  the German low-power/micro HEMP work is Giessen/Airbus/ZARM/DLR (Keller et al. 2015,
  Hey et al. 2015) and is cited instead.
- Liu et al. 2015, "Plume control of a cusped field thruster", IEEE TPS: the DOI I had
  returned 404 at Crossref; it is not cited (the HIT plume papers with verified DOIs are).
- Courtney and Martinez-Sanchez IEPC-2007-39 (DCFT): the electricrocket URL returned 404;
  the DOI-verified AIAA 2008-4631 follow-up is cited instead.
- Lazurenko et al. 2013 (IEPC-2013-339, HEMP QM/FM tests) is cited with the ERPS archive
  URL that the search index served with its abstract; a direct fetch from this machine was
  refused, so treat that entry as "index-verified", not "opened".

### 0.3 The blockers in one paragraph each (from the repository)

**Surrogates.** Two GP surrogates of the 96-design wall-loss screening dataset
(`orbit_wall_loss_geometry_screening_v1`: 512 collisionless launches per design, four
axial cells of 128) were rejected on pre-declared gates. v1 (raw design selectors) had
pooled RMSE 0.0562 against a 0.05 gate, floor-corrected cell RMSE 0.129, and did not beat
ridge (ratio 0.97 against a 2x gate). v2 (31 derived geometry/field features) reached pooled
RMSE 0.0337 (pass) but cell RMSE 0.0904 raw / 0.0836 floor-corrected (fail), ridge on the
same features 0.0334 (ratio 0.99, fail); the learning curve was flat from 30 designs
(0.0455 -> 0.0449 at 50). The per-cell binomial floor at n = 128 is 0.031-0.037; the pooled
floor 0.020. The v2 assessment shows the error concentrated in cell 4 (RMSE 0.142 against
0.059-0.071 for cells 1-3) and coverage failures at saturated cells (truth 128/128).
Earlier, nine L0 surrogate campaigns ended negative or invalidated (v9 learned a closed-form
target: tautology) and ten L1a field-surrogate campaigns ended in development rejection (v10
localised interpolators were worse than the uncorrected coarse solve, 0.342 vs 0.305).

**MDO.** Two preregistered campaigns ran. v1 optimised the operating point of the corrected
L0 model under closure CL-1 `S = prod(1 - p_k)` with independent U[0, 0.45] cusp
probabilities; qLogNEHVI beat LHS 3/3 and NSGA-III 3/3 at 96 evaluations, and the audit
recorded that all eight binding gates are integrity gates (F26) and that 3/3 has a one-sided
sign-test p of 0.125. v2 optimised the 96-design catalogue x operating point with per-cell
Jeffreys-Beta posteriors from the screening counts; the robust front sits on designs
49/50/94, one BO seed stalled at 0.49x the dense reference, and the CL-2 front
(`S = 1 - p_pooled`) shares zero designs with the CL-1 front. The corrected Kornfeld
four-cell power balance has no admissible root for any interior cusp probability, so no
physically closed model links wall loss to performance.

**External validation.** 0 %. GATE-L1 to GATE-L3 are closed by construction. The only
external numbers in the repository are Yeo et al. (2020) model outputs (MDO original
102.7 mN / 36.5 % / 2131 s; PIC 62.8 mN / 15.2 % / 1333 s; MDO modified 61.7 mN / 14.6 % /
1280 s) and the Kornfeld DM9.2 table, which the plasma ledger already used to fix residual
signs (max misfit 1.49e-3), so DM9.2 is spent as calibration data. The PIC steady-state v2
plateau (development, not preregistered) gives I_d 3.44 mA, I_beam 2.29 mA (0.67), 46 %
propellant utilisation at 300 V and 0.0186 mg/s on a 2 mm bore, with dz/lambda_D = 3
at the density peak (under-resolved) and seed-to-seed agreement <= 1.1 %.

## 1. Blocker 1: surrogates on Monte-Carlo (binomial) labels

### 1.1 What the labels are, and what the floor implies

Each label is a binomial count: `s_k` wall hits out of `n_k = 128` launches per cell (512
pooled). The irreducible label standard error is `sqrt(p(1-p)/n)`: 0.040 at p = 0.7 and
n = 128, 0.020 at n = 512, 0.010 at n = 2048. A surrogate's assessment RMSE against
*observed* labels cannot fall below the floor in expectation, and a gate of 0.05 with a floor
of 0.035-0.040 leaves 0.03-0.035 of signal error before it trips, so the v1/v2 cell gate was
never a fair test of the model. The pooled gate (floor 0.020) was, and v2 passed it. That
pattern — pooled pass, cell fail, ridge as good as the GP, learning curve flat beyond 30
designs — is the textbook signature of a label-noise-limited problem, not a model-family
problem (Viering and Loog 2023 review learning curves and their irreducible-error plateaus;
Hestness et al. 2017 show the same "irreducible error region" empirically). The repository's
own diagnosis reached the same conclusion; the literature supports it and says what to do.

### 1.2 State of the art

**Stochastic kriging and heteroscedastic GPs (known, input-dependent noise).**
Ankenman, Nelson and Staum (2010) introduced stochastic kriging: a GP on the *mean* response
with an explicit, estimated intrinsic (replication) variance per design point, and showed
how the intrinsic variance enters the predictor and the design of replications. Binois,
Gramacy and Ludkovski (2018) give the practical heteroscedastic GP (hetGP) with a latent
log-variance GP and replication-aware likelihood, and Gramacy (2020) gives the textbook
treatment of heteroskedastic surrogates; Gramacy and Lee (2012) argue the nugget must be modelled even for deterministic
codes. Our v1/v2 already passed the known binomial variance as fixed `train_Yvar`, which is
stochastic kriging with a known (not estimated) intrinsic variance — the right structure. The
mismatch is that the noise is passed in a *logit* working space via a delta-method variance,
which fails at saturated counts (128/128): the logit of 1.0 is infinite and the
Haldane-Anscombe correction makes the variance arbitrary. The coverage misses in the v2
assessment are exactly at truth = 1.0 cells.

**Binomial-likelihood GPs and Beta regression (probability targets).** The principled
alternative is a GP with a binomial (or Bernoulli) observation model: Williams and Barber
(1998) for the Laplace approximation, Kuss and Rasmussen (2005) for Laplace vs expectation
propagation (EP is better calibrated), Hensman, Matthews and Ghahramani (2015) for scalable
variational inference (what GPyTorch's approximate GPs implement). Rasmussen and Williams
(2005, ch. 3) is the reference. A binomial likelihood handles saturated cells natively (the
posterior over the latent is wide but finite) and turns "floor-corrected RMSE" from a
post-hoc subtraction into a model property. For a parametric baseline, Ferrari and
Cribari-Neto (2004) Beta regression models a proportion with a mean link and a precision
parameter; with counts available a binomial GLM (logit link) on the derived features is the
right *primary* linear model, not ridge on raw proportions.

**Replication versus exploration under a budget.** Binois, Huang, Gramacy and Ludkovski
(2019) address exactly the question "how many replicates at how many sites" for stochastic
simulation experiments: with heteroscedastic noise, an IMSPE-based sequential rule decides at
each step whether to replicate an existing site or explore a new one, and replication is
often optimal where the noise-to-signal ratio is high. Chen and Zhou (2017) give sequential
allocation for stochastic kriging that balances exploration and exploitation of the
intrinsic variance; Chen, Ankenman and Nelson (2012) show that common random numbers (the
same launch seeds across designs) tend to degrade stochastic-kriging prediction of the mean
surface while improving estimates of differences and gradients; the OCBA line (Chen, Lin, Yücesan and Chick
2000) is the ranking-and-selection view of the same trade. For the multi-fidelity variant
— cheap coarse orbits and expensive fine ones — Kennedy and O'Hagan (2000) autoregressive
co-kriging, Giles (2008, 2015) multilevel Monte Carlo, and the survey by Peherstorfer,
Willcox and Gunzburger (2018) give the allocation rules; MLMC puts almost all samples at the
cheapest level when the level-to-level correction has small variance, which is our case
(N -> 2N changes <= 0.0059 in P(wall)).

**Variance reduction for test-particle probabilities.** Caflisch (1998) covers antithetic
variables, control variates, stratification and quasi-Monte Carlo (QMC) with their variance
formulas; QMC on the launch phase space (position, pitch angle, gyrophase) replaces the
O(n^-1/2) binomial error with a faster-decaying discrepancy error for smooth dependence, and
stratification by cell is already in place. Brown, Cai and DasGupta (2001) justify the Wilson
interval used throughout the bundles and warn against the Wald interval near p = 0, 1.

**Non-stationary, mixed-variable and categorical inputs.** Roustant et al. (2020) group
kernels and Zhang, Tao, Chen and Apley (2020) latent-variable GPs (LVGP) handle categorical
inputs by learning an embedding rather than treating levels as exchangeable; Pelamatti et al.
(2019) benchmark mixed-variable GP kernels for constrained optimisation; Gramacy and Lee
(2008) treed GPs and Sauer, Gramacy and Higdon (2023) deep GPs (building on Damianou and
Lawrence 2013) handle step discontinuities and regime changes. Our v1 diagnosis (signal on
`stage_count_selector` and `exit_length_fraction`, a step map) is the textbook case for
treed/LVGP models — but v2 showed that once physically meaningful features are used the
model family no longer matters (GP = ridge = trees within noise), so the input representation
was the real fix and non-stationarity is now secondary.

**When a linear model is the right surrogate, and how to gate.** Nadeau and Bengio (2003)
and Varma and Simon (2006) quantify the variance and optimism of small-sample
cross-validation estimates; Cawley and Talbot (2010) show that selecting among several models
on a small selection set over-fits the selection criterion by amounts comparable to the
differences between methods; Demšar (2006) gives the non-parametric tests appropriate for
comparing methods over multiple sets. Kapoor and Narayanan (2023) catalogue leakage patterns
(including "no test set", "pre-processing on train+test", and "temporal leakage") that
produce tautological or optimistic results; the L0 v9 tautology is their "illegitimate
features" case (inputs that encode the target). Saltelli et al. (2010) is the standard reference for the
variance-based sensitivity estimators the repository keeps closed until surrogate error is
propagated.

### 1.3 Pitfalls documented in the literature that match our record

1. **Gates that ignore the label-noise floor.** A cell gate of 0.05 with a floor of 0.035-0.040
   tests the labels, not the model (section 1.1). Stochastic-kriging practice reports and gates
   against the *latent mean* error, estimated by replication (Ankenman et al. 2010; Binois et
   al. 2018).
2. **"Beat the baseline by 2x" when the baseline shares the informative features.** v2 made
   this gate unmeetable by construction (ridge 0.0334 vs GP 0.0337). Cawley and Talbot (2010)
   and Boulesteix, Lauer and Eugster (2013) treat method comparison as its own inference
   problem; a fixed multiplicative margin is not one.
3. **Method selection on 10 designs.** v1 selected `botorch-icm-logit`, the shakedown
   `botorch-stgp-direct`; Cawley and Talbot (2010) predict exactly this instability.
4. **Delta-method logit noise at saturated counts.** Coverage misses at truth = 1.0 (v2
   assessment, cells 2-4) are a likelihood-misspecification symptom, not a calibration one;
   the 1.86x (v2) and 3.34x (v1) variance inflations are absorbing it.
5. **Tautological targets.** L0 v9 (analytic leading term equals the target); the repository's
   `no_tautology` gate is the right response and matches Kapoor and Narayanan (2023).
6. **Interpolating a discontinuous map.** L1a field-surrogate v10's localised interpolators
   underperforming the coarse solve is consistent with Gramacy and Lee (2008): stationary
   local interpolation across regime boundaries is worse than the low-fidelity model itself.

### 1.4 Recommendations mapped to experiments

**Screening v2 (label precision first).** Two budget options, both preregistered before any
launch; the figures use the v1 rate (100,352 orbits in 95 min on 12 workers under load):

- Option A — *targeted replication* (Binois et al. 2019 rule, frozen a priori): keep all 96
  designs and the v1 partition; add launches only to cells whose current Wilson 95 % width
  exceeds 0.10 (i.e. non-saturated cells; 73 of 96 designs have a 128/128 cell that gains
  nothing from replication), bringing each such cell to n = 512. With ~2.4 non-saturated
  cells per design this is ~88 k extra launches at N (about 0.9x the 100,352 orbits v1
  integrated, since v1 ran every design at N and 2N; ~85 min at v1's rate) and lowers the
  cell floor from 0.040 to 0.020 where it matters (cells 1 and 4, which carry the v2 error).
- Option B — *fewer designs, more replicates*: 2048 launches (512 per cell) on 40 designs
  chosen to include the frozen 16 assessment and 10 extrapolation designs plus a maximin
  subset of the rest (~82 k launches at N, ~0.8x v1's orbit count). Per the learning curve
  (flat from 30 designs)
  this loses little signal and quarters the label variance. Binois et al. (2019) is the
  citation for preferring replication when noise dominates.

Either way: run at the accepted N time step with a 10 % 2N control sample (Giles 2015 MLMC
logic; N -> 2N differences <= 0.0059 are below the target floor); use a scrambled-Sobol
launch set stratified by cell and pitch-angle sign (Caflisch 1998); keep the *same* launch
set across designs only if the estimand is a between-design difference (MDO ranking), and
use independent sets if the estimand is the per-design mean surface (Chen et al. 2012).

**Surrogate v3 (model and gates).**

- Primary model declared a priori: a binomial-likelihood GP (variational or EP; Hensman et
  al. 2015; Kuss and Rasmussen 2005) on the v2 derived features, per cell, with the cusp/null
  distances in pitches as inputs; a binomial GLM (logit) on the same features as the declared
  *parametric* candidate. The GP is a candidate, not the hypothesis.
- Gates in probability units against the floor computed from the actual `n`: cell RMSE
  <= 1.5x the per-cell binomial floor (0.030 at n = 512; 0.061 at n = 128 if v3 must run on v1
  labels); pooled RMSE <= 1.5x pooled floor; 90 % interval coverage in [0.85, 0.97] computed
  with the binomial predictive, not a scaled Gaussian.
- Replace "2x best baseline" by a *reliability ceiling* check: split each design's launches
  into two independent halves, compute the split-half correlation of the per-cell
  proportions across designs; the attainable R^2 of any surrogate is bounded by that
  reliability, and a surrogate is "useful" if it recovers >= 70 % of it. This is an
  observation-level, model-free ceiling and cannot be gamed by the model family.
- Method selection: nested cross-validation on the fit + method-selection roles (Varma and
  Simon 2006), at least 26 designs in the selection role, or none at all (declare one model).
- Keep the `no_tautology`, single-use-label and frozen-partition gates exactly as they are.

**L0 surrogates.** None. The v9 audit's conclusion stands: the useful object is a
*discrepancy* surrogate between L0 and a higher-fidelity output (Kennedy and O'Hagan 2000
AR1 form; Peherstorfer et al. 2018), which needs a higher-fidelity output that does not yet
exist as accepted evidence.

## 2. Blocker 2: MDO under closure (model-form) uncertainty

### 2.1 State of the art in the tools we already run

BoTorch (Balandat et al. 2020) provides the Monte-Carlo acquisition framework; qEHVI
(Daulton, Balandat and Bakshy 2020) and its noisy-observation form qNEHVI (Daulton, Balandat
and Bakshy 2021) are the parallel hypervolume-improvement acquisitions we use, and the
LogEI family (Ament et al. 2023) fixes their vanishing-gradient pathology (`qLogNEHVI`).
Constraints at scale: SCBO (Eriksson and Poloczek 2021). High-dimensional multi-objective:
MORBO (Daulton, Eriksson, Balandat and Bakshy 2022). Frazier (2018) is the tutorial.

**Robust and risk-aware BO.** Cakmak, Astudillo, Frazier and Zhou (2020) define Bayesian
optimisation of risk measures (VaR/CVaR of the objective under input uncertainty) with a
one-step lookahead acquisition; Daulton, Cakmak et al. (2022) extend to multiple objectives
under input noise with MARS (multi-objective robust acquisition via random scalarisations),
which optimises a CVaR-type set-based risk measure and is implemented in BoTorch. Our v1/v2
CVaR(0.25) over a frozen 64-row QMC sample is a fixed-sample plug-in of the same estimand;
MARS would give the acquisition itself the risk measure instead of the evaluation chain.

**Mixed categorical-continuous BO.** Ru et al. (2020, CoCaBO) combine bandits over categories
with GP-BO over the continuous part; Wan et al. (2021, Casmopolitan) use trust regions and a
mixed kernel; Daulton, Wan et al. (2022) optimise the acquisition over discrete/mixed spaces by
probabilistic reparameterisation with exact gradients; Pelamatti et al. (2019), Roustant et
al. (2020) and Zhang et al. (2020, LVGP) give the kernels. The v2 stall (seed 101 never found
design 49 because the categorical kernel treats 96 designs as exchangeable) is the known
weakness of an exchangeable categorical kernel and the LVGP/descriptor remedy is the
standard fix: give the GP continuous design descriptors (the screening P(wall) per cell, the
v2 derived features) so that nearby designs share information.

### 2.2 Structural (model-form) uncertainty: what the literature says to do when closures disagree

Kennedy and O'Hagan (2001) introduced the explicit model-discrepancy term `delta(x)` between a
simulator and reality and warned that calibrating parameters without it biases them;
Brynjarsdóttir and O'Hagan (2014) show that even *with* a discrepancy term, physical
parameters are identifiable only with prior information about the discrepancy's form. Roy
and Oberkampf (2011) place model-form uncertainty as an *epistemic* interval (p-box) that is
never averaged away; Riley and Grandhi (2011) quantify model-form uncertainty from a set of
competing models by adjustment factors; Tebaldi and Knutti (2007) is the canonical
discussion of multi-model ensembles and why "model democracy" (equal weights) is a choice
that must be argued, not assumed. In optimisation, Beyer and Sendhoff (2007) survey robust
optimisation, Deb and Gupta (2006) define robust multi-objective fronts, and Ide and
Schöbel (2016) survey set-based robustness concepts for *uncertain multi-objective*
problems — minmax, minmax-regret, highly robust (efficient under every scenario) and flimsily
robust (efficient under at least one) — which is the vocabulary needed when two closures give
disjoint Pareto sets.

The electric-propulsion literature has an exact analogue of our closure problem. Jorns (2018)
learned a data-driven closure for the anomalous electron collision frequency; Marks and Jorns
(2023) then showed that closures calibrated against empirically inferred profiles *diverge
from those profiles when implemented self-consistently* in the fluid code, because of
non-linearity, time-averaging artefacts and non-uniqueness of the inferred collision
frequency; Mikellides and Lopez Ortega (2019) document the same challenge for
first-principles anomalous-resistivity models. Hara (2019), Boeuf (2017) and Kaganovich et
al. (2020) review the modelling landscape. For cusped-field devices specifically, Kahnfeld et
al. (2019) review a decade of Greifswald PIC on the HEMP-T and show that the potential
structure and the losses are set by kinetic electron confinement plus sheaths — precisely the
physics a collisionless prescribed-field test particle does not have.

The applied lesson for us: a closure that identifies a collisionless wall-hit probability
with a per-cusp survival factor (CL-1) is a *scenario*, not a model; the v1 audit already
showed that feeding the v4 P2-field probabilities into it yields a beam of ~1e-7 of the ions.
Under Ide and Schöbel's taxonomy, the reportable objects are: the per-closure fronts, the
*highly robust* set (designs efficient under every declared closure — empty for v2 since the
CL-1 and CL-2 fronts are disjoint), the *flimsily robust* set (the union), and the minmax
regret across closures. Reporting a single "robust front" under one closure is what the
literature says not to do, and the v2 protocol's forbidden-readings list already says the
same.

### 2.3 Pitfalls: hypervolume, seeds, integrity gates

- **Hypervolume as the only indicator.** Zitzler et al. (2003) prove which indicators are
  Pareto-compliant and show that no single unary indicator can rank fronts in general;
  Ishibuchi et al. (2018) show the hypervolume ranking depends on the reference point and
  give a rule for choosing it; Audet et al. (2021) survey and classify the performance
  indicators and recommend reporting cardinality, convergence (e.g. IGD+) and distribution
  indicators alongside HV.
- **Seeds.** Derrac et al. (2011) give the non-parametric protocol (Wilcoxon/Friedman with
  post-hoc corrections) and its sample-size needs; COCO (Hansen et al. 2021) uses 15 instances
  per function as its default; Bartz-Beielstein et al. (2020) list best practice for
  optimisation benchmarking; Henderson et al. (2018) showed in deep RL that 5 seeds can produce
  contradictory rankings. Three seeds report counts; they cannot report significance (the
  v1 audit's F17 says exactly this).
- **Integrity gates mistaken for efficacy.** Pawel, Kook and Reeve (2024) demonstrate that in
  comparative simulation studies any method can be made to "win" by choosing scenarios,
  metrics and stopping rules after seeing results; Nießl et al. (2022) quantify the
  over-optimism from the multiplicity of design/analysis options; Boulesteix et al. (2013)
  call for *neutral* comparison studies designed by parties without a stake in the outcome.
  Our v1 audit's F26 ("acceptance = pipeline integrity, efficacy statements reported-not-
  binding") is the correct labelling; the risk is downstream readers treating `accepted_result`
  as "BO works".

### 2.4 State of practice in electric-propulsion design optimisation

The direct lineage is surrogate-assisted NSGA-II over a 0-D power-balance model plus
magnetostatics: Fahey, Muffatti and Ogawa (2017; three objectives, five decision variables,
population 64 x 40 generations then surrogate-only 100 x 100), Muffatti and Ogawa (ISTS
2017-b-32; the study this repository audits), Yeo and Ogawa (2019; magnet thickness), Yeo,
Ogawa, Matthias, Kahnfeld and Schneider (2020; MDO points re-evaluated by Greifswald PIC —
the PIC gave 62.8 mN against the original MDO's 102.7 mN, and the PIC-informed MDO 61.7 mN),
Yeo and Ogawa (2022; surrogate-assisted EA in JPP), Yeo and Ogawa (2023; Monte-Carlo
uncertainty propagation through surrogates), and Yeo, Gadisa, Ogawa and Bang (2024; the same
methodology applied to FEEP). Matthias et al. (2019) is the PIC paper on the optimised
HEMP design. Outside that group: Lewerentz and Schneider (2023) optimise the HEMP magnetic
configuration with a simplified objective; Puca, Panelli and Battista (2024) build a
preliminary-design tool from analytical models plus a 2-D magnetostatic solver. The general
MDO framing is Martins and Lambe (2013); uncertainty-based MDO for aerospace is reviewed by
Yao et al. (2011).

What none of these reports: seed statistics for the optimiser, a preregistered protocol, an
integrity/efficacy distinction, or an experimental measurement of an optimised design. The
Yeo et al. (2020) MDO-vs-PIC gap (a factor 1.6 in thrust) is the best published estimate of
the model-form error of the 0-D chain this project inherited, and it is *cross-model*, not
experimental.

### 2.5 Recommendations mapped to experiments

- **MDO v3 design representation**: LVGP or descriptor kernel (Zhang et al. 2020) with the
  four screening P(wall) and the v2 top features as continuous descriptors; keep the
  exhaustive categorical candidate stage as a baseline. Expect the seed-101 stall to
  disappear; test it with >= 10 seeds.
- **Seeds and indicators**: >= 10 seeds per optimiser (15 if the budget allows, matching
  COCO); report HV *and* IGD+ *and* the additive epsilon indicator with bootstrap CIs over
  seeds; fix the reference point by Ishibuchi et al.'s rule and record it in the protocol;
  paired Wilcoxon only at n >= 10, effect sizes always.
- **Closure as scenario, reported by Ide-Schöbel sets**: fronts under CL-1, CL-2 and any
  further declared closure; the highly-robust (intersection), flimsily-robust (union) and
  minmax-regret sets; never a pooled or averaged front. The current disagreement (Jaccard 0)
  *is* the result and should be stated as the model-form uncertainty of the chain.
- **An efficacy gate if efficacy is to be claimed**: e.g. "median final HV over >= 10 seeds
  >= 0.9x the dense reference at the fixed budget"; otherwise keep the wording "integrity
  only", as v2 does.
- **A physics-bearing closure before any design recommendation**: the only route in the
  repository is a closure calibrated from the PIC plateau (wall ion flux per cusp, beam
  fraction, utilisation) — with Marks and Jorns (2023) as the warning that a calibrated
  closure may not reproduce its calibration data when run self-consistently, so it must be
  re-checked against a second PIC operating point before use.

## 3. Blocker 3: external validation (currently 0 %)

### 3.1 What a defensible validation protocol looks like

ASME V&V 20-2009 (reaffirmed 2021) defines the comparison error `E = S - D` between
simulation and data at a validation point and the validation uncertainty
`u_val = sqrt(u_num^2 + u_input^2 + u_D^2)`; the model is characterised by `E +/- u_val`, and
the standard is explicit that it says nothing about points other than the validation points.
Oberkampf and Roy (2010) and Oberkampf and Trucano (2002) are the framework texts (code
verification, solution verification, validation, prediction; the *validation hierarchy* from
unit problems to complete systems). Oberkampf and Barone (2006) set requirements for
validation metrics (account for both numerical and experimental uncertainty; do not collapse
to pass/fail; be a metric); Ferson, Oberkampf and Ginzburg (2008) define the *area metric*
between the simulation's predictive distribution and the empirical distribution of the data
and apply it to a challenge problem with sparse data — the case most like ours; Liu et al.
(2011) compare validation metrics. Bayarri et al. (2007) give the Bayesian framework that
separates calibration from validation and Trucano et al. (2006) spell out why *calibration
data cannot be validation data* and how sensitivity analysis sits between them. Roy and
Oberkampf (2011) integrate aleatory and epistemic uncertainty into the validation statement;
the National Research Council (2012) report is the consensus reference for VVUQ practice; Oberkampf, Pilch and
Trucano (2007) give the Predictive Capability Maturity Model and NASA-STD-7009A (2016) the
Credibility Assessment Scale (eight factors scored 0-4, overall = minimum) — both are
ready-made claim-matrix structures.

For plasma physics specifically, Terry et al. (2008), Greenwald (2010) and Holland (2016)
define the *primacy hierarchy* (how many steps of inference separate a measured quantity
from the compared one), insist that validation metrics carry both uncertainties, and warn
that agreement on a derived, high-primacy quantity (e.g. total thrust) can hide disagreement
on the low-primacy ones (profiles). For the PIC kernel itself, the LANDMARK-style benchmarks
(Charoy et al. 2019, axial-azimuthal; Villafana et al. 2021, radial-azimuthal) are the
community precedent for *code-to-code verification* of ExB PIC before validation.

Preregistration of the validation cases (Nosek et al. 2018) with the metric, the tolerance
and the withheld points declared before the comparison is the part the repository's
`validation` contracts already anticipate.

### 3.2 Published HEMP / cusped-field datasets one could validate against

| Source | Device and scale | Quantities reported (as published) | Uncertainty reported? |
|---|---|---|---|
| Kornfeld, Koch, Harmann 2007 (IEPC-2007-108) (full text read) | Thales HEMP-T DM3a ... DM9.2 (3050 class) and HEMP-T 30250 | Evolution table: DM3a 24 mN maximum thrust and total efficiency up to about 30 % (ion energy distribution shown at 500 V); DM6 129 mN, 2679 s anode Isp, 78 % thermal efficiency, 37 % anode efficiency; DM9.2 = HEMP-T 3050 column; thermal losses ~15 % located at anode and cusps; hollow-cone beam; power-balance model matched to observed thermal losses | No numerical uncertainties |
| Koch, Harmann, Kornfeld 2007 (IEPC-2007-110) | HEMP-T 3050 (50 mN / 1500 W / 3000 s nominal), HEMP-T 30250 | Development status, operating and performance characteristics | No |
| Koch et al. 2011 (IEPC-2011-236) (full text read) | HEMP-T 3050 module at SmallGEO point (1000 V, 1380 W) | RPA ion-energy distribution vs angle: main peak ~985 eV (= Ua minus ~15 V cathode-to-ground), peak ion density at 20 deg, negligible above 50 deg, no ions on axis; broad < 150 eV population (CEX + exit-cusp ionisation); thermal and acceleration efficiencies ~0.85-0.9 from beam integration; direct thermal measurement ~0.85; HEMPT 30250 demo 333 mN at 3200 s and 51 % at 10 kW | No |
| Koch et al. 2011 (AIAA 2011-6086) | HEMPT ion propulsion system for SmallGEO | System-level performance | No |
| Genovese et al. 2011 (IEPC-2011-141) | HEMPT modules for SmallGEO | Endurance testing | Not checked (metadata only) |
| Lazurenko et al. 2013 (IEPC-2013-339; index-verified) | HTM QM + 4 FM | Nominal 1000 V / 1.38 A (> 4 h); characterisation 850-1050 V, 0.3-1.45 A; thrust >= 44 mN; total Isp > 2400 s; plasma beam characterisation | Not in abstract |
| Keller et al. 2015 (IEEE TPS) | mu-HEMPT (Giessen/Airbus), geometry systematically varied, anode material varied | 32-cup Faraday array (330 deg) + 3-grid RPA; indirect thrust: 50 muN at 600 V (Isp 230 s), 180 and 360 muN at Isp 610 and 860 s; beam profile and ion acceleration vs geometry | Not in abstract |
| Hey et al. 2015 (IEEE TPS; J. Phys. Conf. Ser.) | Same mu-HEMPT on a double-pendulum thrust balance | Direct thrust, 0.1 muN resolution in 1e-3-1 Hz | Yes (balance noise) |
| Neumann 2017 (Procedia Eng.) | DLR Göttingen EP test facility diagnostics | Facility/diagnostic capability | n/a |
| Ma et al. 2015 (Vacuum) | HIT variable-magnet-length CFT | Thrust, efficiency, plume angle vs stage-length ratio; optimum ratio identified | Not in abstract |
| Hu et al. 2016 (J. Phys. D) | HIT multi-cusped field thruster, magnet length varied | Performance vs magnet length | Not in abstract |
| Hu et al. 2016 (AIP Advances) | Four thrusters with different magnet-ring outer diameters | Thrust and anode efficiency vs in-channel field strength at several anode voltages (weaker field -> higher thrust and efficiency) | Not in abstract |
| Hu et al. 2015 (Phys. Plasmas) | Five exit shielding rings | Electron current, propellant and current utilisation vs plume-region field; high/low current modes | Not in abstract |
| Hu et al. 2016 (Phys. Plasmas) | HIT MCFT plume | Faraday probe + RPA: ion energy distribution vs angle in both current modes | Not in abstract |
| Wu et al. 2015 (Phys. Plasmas) | Multi-annulus anode | Radial anode current-density distribution | Not in abstract |
| Liu et al. 2014 (Phys. Plasmas); Zhao et al. 2014 (J. Phys. D) | HIT PIC-MCC | Electric-field formation and field-strength effects (model output) | n/a |
| Liu, Zeng, Yu, Huang 2019 (Vacuum) | Low-power HEMP thruster, channel length varied | Channel-length effect on performance | Not in abstract |
| Courtney, Lozano, Martinez-Sanchez 2008 (AIAA) | MIT DCFT | Performance characterisation | Not checked |
| Gildea, Martinez-Sanchez, Nakles, Hargus 2009 (IEPC-2009-259) | MIT DCFT plume | Faraday cup + RPA ion current density and energy vs angle, two operating modes, cathode coupling | Not checked |
| Matlock et al. 2010 (AIAA 2010-7104) | DCFT with magnetic collar | Near-field Langmuir and emissive probes; peak ion current at 30-35 deg; up to 4 kG on axis | Not checked |
| Gildea et al. 2010 (AIAA 2010-7014) | DCFT | Low-frequency oscillations | Not checked |
| Daspit, Lozano, Martinez-Sanchez 2011 (AIAA) | DCFT on a calibrated horizontal accelerometer stand | Direct thrust | Not checked |
| MacDonald et al. 2011 (J. Phys. D); MacDonald et al. 2012 (J. Appl. Phys.) | DCFT and a cylindrical cusped-field thruster | LIF ion velocity in channel and plume; plasma potential; high- vs low-current modes | Yes (LIF) |
| Gildea et al. 2013 (J. Propul. Power) | Low-power cusped-field thruster | Erosion measurements (wall-loss footprint) | Yes |
| Raitses and Fisch 2001; Smirnov et al. 2002, 2004; Raitses et al. 2007 | Princeton cylindrical Hall thruster (CHT) | Thrust, efficiency, probe profiles at 100 W class | Partly |
| Leung, Hershkowitz, MacKenzie 1976 (Phys. Fluids) | Laboratory cusp (not a thruster) | Plasma confinement and leak width in localized cusps | Yes |
| Yeo et al. 2020 (JSR); Matthias et al. 2019 (CPP) | Optimised downscaled CFT (model only) | PIC thrust/efficiency/Isp for MDO points (cross-model) | n/a |

Caveats on the table: Koch et al. (2011) state that the classical CHT "differs substantially"
from the HEMPT while the DCFT is "a close copy" — so Princeton CHT data are a methodology
reference only, and DCFT/MIT and HIT data are the nearest same-class experiments. The
original ISTS 2017 study and Fahey et al. (2017) are simulation-only; there are no device
measurements from the original study.

### 3.3 Which of our quantities could be compared to which measurements

| Our quantity (level, status) | Nearest measured counterpart | Operating point | What is missing |
|---|---|---|---|
| Test-particle P(wall) per cell (`GATE-WALL-LOSS-V4`, screening v1) | None directly. Indirect only: thermal-loss fraction localised at anode and cusps (~15 %, Kornfeld et al. 2007), erosion footprint (Gildea et al. 2013), leak width in a laboratory cusp (Leung et al. 1976) | HEMP-T at 1 kV; DCFT at 200-500 V | A collisionless, field-free electron orbit is several primacy steps from any of these (Greenwald 2010); a comparison would need a model of what fraction of the wall-hit electrons carries the thermal load. Not a validation target as it stands. |
| L0 thrust / Isp / efficiency (`CLM-005..008`, hypothetical inputs) | HEMP-T evolution table (Kornfeld 2007), HTM 44 mN / > 2400 s at 1000 V, 1.38 A (Lazurenko 2013) | 850-1050 V, 0.3-1.45 A | Inputs are hypothetical; the DM9.2 column is already used for calibration of the plasma ledger, so only *other* points (DM3a, DM6, HTM range) are admissible as validation, and their uncertainties are unpublished. |
| PIC plateau I_d(Ua, mdot), beam fraction 0.67, utilisation 0.46 (development) | HTM I-V envelope (0.3-1.45 A at 850-1050 V); mu-HEMPT thrust vs Ua (Keller 2015: 50-360 muN; the 50 muN point at 600 V); HIT low-power HEMP (Liu 2019) | Ours: 300 V, 3.4 mA, 2 mm bore | Scale mismatch with HEMP-T 3050 (1 kV, 1.4 kW); the mu-HEMPT and HIT low-power devices are the right class, but their geometries must be modelled, which needs the published dimensions or author contact. |
| PIC exit ion-energy distribution and angular structure (available from the plateau run at low cost) | RPA peak at Ua minus cathode-to-ground voltage, peak at 20 deg, negligible > 50 deg, hollow cone, broad < 150 eV tail (Koch et al. 2011); HIT RPA in both current modes (Hu et al. 2016); DCFT RPA (Gildea et al. 2009) | 1000 V (HEMP-T); 200-500 V (DCFT/HIT) | Our run is 300 V on a different geometry: the comparison is *same-class behaviour* (peak location relative to Ua, hollow-cone divergence), not a device validation; still the cheapest first E +/- u_val exercise. |
| PIC density / potential maps (development) | LIF ion velocity and plasma potential in a DCFT and a cylindrical cusped-field thruster (MacDonald et al. 2011, 2012); emissive-probe potentials in DCFT near field (Matlock et al. 2010) | DCFT 200-500 V | No published interior maps for HEMP-T except PIC (Matyash et al. 2010; Kahnfeld et al. 2019), which makes a code-to-code comparison the realistic first step. |
| Geometry trends: P(wall) vs stage pitch / stage count (screening v1: pitch +0.36, stages -0.31 Spearman) | Performance vs magnet length (Ma et al. 2015; Hu et al. 2016 J. Phys. D), vs in-channel field strength (Hu et al. 2016 AIP Adv.), vs channel length (Liu et al. 2019) | HIT devices, 100-400 V class | A *trend* validation (sign and monotonicity) is possible once a geometry -> performance chain exists; today the chain stops at wall-hit probability. |

### 3.4 Pitfalls

1. **Comparing collisionless test-particle results to plasma measurements.** The HEMP-T's
   potential structure and losses are set by kinetic electron confinement, sheaths and the
   exit-cusp electron cloud (Koch et al. 2011; Kahnfeld et al. 2019); the test-particle
   P(wall) has no electric field and no collisions. Any such comparison would be at primacy
   distance >= 3 (Greenwald 2010) and is not a validation.
2. **Calibrating on the validation set.** DM9.2 fixed the plasma-ledger signs; Trucano et al.
   (2006) and Bayarri et al. (2007) are explicit that it is then spent. The same applies to
   any PIC-calibrated closure: the PIC operating point used to fit `p_k` cannot be the one
   used to validate it.
3. **Uncertainty-free comparisons.** Most HEMP conference data carry no numerical
   uncertainty; ASME V&V 20 requires `u_D`. Where it is absent the protocol must *declare* an
   assumed `u_D` and mark the comparison as conditional, or use the area metric with the
   spread of repeated measurements where those exist (Ferson et al. 2008).
4. **Using another model's output as truth.** The Yeo et al. (2020) PIC row and the
   Greifswald PIC maps are cross-model evidence (the repository already labels them so).
5. **Facility effects.** The < 150 eV ion population in Koch et al. (2011) is attributed to
   CEX with background gas and exit-cusp ionisation; Duras et al. (2017) needed a CEX
   post-processing step to reproduce 1 m RPA data from PIC. Background pressure belongs in
   `u_input`.
6. **Numerical uncertainty left out of `u_val`.** The plateau run has dz/lambda_D = 3 at the
   peak; solution-verification error must be in `u_num` before any `E` is quoted (Oberkampf
   and Trucano 2002).

### 3.5 Recommendations mapped to experiments

- **validation v0 (code-to-code, opens nothing but is a prerequisite)**: reproduce a
  published Greifswald HEMP PIC case (Matyash et al. 2010; Matthias et al. 2019 for the
  optimised CFT design; Kahnfeld et al. 2019 for the model description) on the repository's
  `pic2d` and report the exit potential drop, plateau density and beam fraction differences
  with both codes' stated numerical parameters — the LANDMARK precedent (Charoy et al. 2019;
  Villafana et al. 2021) is the template for what to tabulate.
- **validation v1 (first E +/- u_val, GATE-L1 candidate at most)**: quantity = location of the
  main ion-energy peak relative to the applied anode voltage and the angle of peak ion
  current density, from the PIC plateau at 300 V; referent = Koch et al. (2011) HEMP-T 3050
  at 1000 V (peak at ~Ua - 15 V, 20 deg) and Hu et al. (2016) HIT RPA data; `u_num` from a
  Δz refinement (the dz/lambda_D = 3 run is not sufficient), `u_input` from n_g and Q_in,
  `u_D` declared (assume ±5 % of Ua for the peak, ±5 deg for the angle, stated as an
  assumption). Declare in advance that this tests same-class behaviour, not the device.
- **validation v2 (thrust-voltage curve)**: model the mu-HEMPT of Keller et al. (2015) or the
  low-power HEMP of Liu et al. (2019) once dimensions are secured; compare thrust vs Ua at
  fixed flow (Keller: 50-360 muN, 50 muN at 600 V) with the area metric across their reported
  operating points; withhold one operating point.
- **validation v3 (geometry trend)**: once a geometry -> performance chain exists, compare
  the sign and monotonicity of efficiency vs magnet/stage length against Ma et al. (2015) and
  Hu et al. (2016) and vs in-channel field strength against Hu et al. (2016, AIP Advances).
- **Claim-matrix structure**: map each gate kind to the NASA-STD-7009A factors
  (verification, validation, input pedigree, results uncertainty, results robustness) and
  score 0-4; the current state is verification 3-4, validation 0, input pedigree 1
  (hypothetical inputs), which is an honest one-line summary for the paper.

## 4. Blocker 4: reproducibility and preregistration of computational campaigns

### 4.1 Precedent

Preregistration and Registered Reports: Nosek et al. (2018) set out the logic (separating
prediction from postdiction); Chambers and Tzavella (2021) review Registered Reports (stage-1
protocol review before data, stage-2 review of results regardless of outcome); Hardwicke and
Wagenmakers (2023) give a balanced account of what preregistration can and cannot do.
Preregistering *computational* work: Crüwell and Evans (2021) give a preregistration template
for cognitive-model applications (model specification, parameter estimation, model
comparison and robustness declared up front); Morris, White and Crowther (2019) define the
ADEMP structure (Aims, Data-generating mechanisms, Estimands, Methods, Performance measures)
for simulation studies and Siepe et al. (2024) turn it into a preregistration and reporting
template; Cockburn et al. (2020) argue for preregistration in empirical computer science; the
NeurIPS 2020 Workshop on Pre-registration in Machine Learning (Bertinetto et al., eds, PMLR
148, 2021) published proposals reviewed before experiments were run; Pineau et al. (2021)
report the NeurIPS reproducibility programme (code submission, checklist, reproducibility
challenge) and Heil et al. (2021) propose bronze/silver/gold reproducibility standards.

Neutral comparison and over-optimism: Boulesteix et al. (2013), Nießl et al. (2022), Pawel et
al. (2024) — the last shows concretely how post-hoc choices in comparative simulation studies
let any method win.

Artefacts and packaging: FAIR principles (Wilkinson et al. 2016); RO-Crate (Soiland-Reyes et
al. 2022) for packaging a research object with provenance metadata; Barba (2018) for the
reproducible/replicable terminology; Peng (2011) and Stodden et al. (2016) for the
reproducibility spectrum and journal-level requirements; Sandve et al. (2013) for the ten
rules; Ivie and Thain (2019) for a survey of reproducibility in scientific computing.

Credibility scales: NASA-STD-7009A (2016) and the PCMM (Oberkampf, Pilch and Trucano 2007)
are the engineering-side equivalents of a claim matrix with graded evidence.

### 4.2 Pitfalls we should watch

1. **Garden of forking paths in gate tuning.** Gelman and Loken (2014) and Simmons, Nelson
   and Simonsohn (2011): analysis choices made after seeing data inflate false positives even
   without fishing. Our v1 -> v2 -> (proposed) v3 surrogate gate changes are each justified,
   but a reader cannot distinguish justified from data-driven unless the *gate genealogy*
   (which gate changed, when, on what evidence, before or after which result) is recorded in
   the protocol. Every v3 gate proposed in section 1.4 is therefore stated here, before any
   v3 data exist.
2. **Shakedown contamination.** Sagarin, Ambler and Lee (2014) on "peeking" and Albers and
   Lakens (2018) on biased pilot-based decisions apply directly: shakedowns that see the real
   labels (v1 surrogate shakedown on the real dataset with a shakedown partition; MDO v1 five
   shakedowns before the freeze) can leak into method choice. The v1 surrogate's method
   instability (shakedown picked a different model) is the visible symptom. Remedy: shakedown
   on synthetic or permanently excluded designs, log every shakedown outcome in the bundle,
   and declare method-selection rules that cannot be edited after a shakedown.
3. **Multiplicity of analyses across versions.** Nine L0 and ten L1a surrogate versions, plus
   two wall-loss surrogates, is a multiple-comparison problem: the probability that *some*
   version passes by chance rises with the count. The repository's practice of recording
   every failure is the right protection; a v3 protocol should state the family-wise count.
4. **Integrity vs efficacy labelling.** Already handled in MDO v2; must propagate to any
   surrogate acceptance ("predictor usable" is an efficacy statement and needs an
   outcome gate).
5. **Preregistration that only lives in Git.** Git history can be rewritten; a third-party
   timestamp (a public registry entry or a signed tag mirrored outside the repository)
   is what Nosek et al. (2018) mean by preregistration.

### 4.3 Recommendations

- Adopt ADEMP fields (Morris et al. 2019; Siepe et al. 2024) as mandatory protocol sections
  for every campaign: aims, data-generating mechanism (here: field solver, launch
  distribution, closure), estimands, methods, performance measures — most already exist under
  other names in `protocol.json`.
- Register each `preregister ...` commit hash in an external registry (or mirror signed tags
  to a second host) at the moment of freezing; record the registry receipt in
  `authorities.json`.
- Package each accepted bundle as an RO-Crate (Soiland-Reyes et al. 2022) with the existing
  manifest as the payload; the metadata cost is small and it makes the bundles FAIR.
- Add a `gate_genealogy` block to every protocol listing predecessor gates and the evidence
  that motivated each change.
- For optimiser comparisons, adopt the neutral-comparison design (Boulesteix et al. 2013):
  declare the seed count, indicators, reference point and statistical test in the protocol,
  and report Pawel et al.'s (2024) checklist items.
- Score every admitted result on the NASA-STD-7009A factors in the claim matrix; the
  validation factor at 0 is then visible in every table rather than in a footnote.

## 5. Summary table

| Blocker | Change | Effort (planning estimate) | Risk |
|---|---|---|---|
| Surrogate labels are noise-limited | Screening v2 with targeted replication (n = 512 in non-saturated cells) or 2048 launches on 40 designs; QMC launch sets; N-step with 2N control | 1 d protocol + ~1.5-3 h CPU + 1 d record/audit | Budget under contention (v1 ran 35 % over projection); allocation rule must be frozen before seeing v2 counts |
| Surrogate gates ignore the floor and compare against an equal baseline | v3 gates: RMSE <= 1.5x binomial floor from actual n; binomial-predictive coverage; split-half reliability ceiling instead of 2x baseline; primary model declared a priori | 2-3 d (binomial-likelihood GP + GLM, tests, protocol) | A third rejection if screening v2 is not run first; GP tooling (variational binomial) not yet in repo |
| Delta-method logit noise fails at saturated cells | Binomial-likelihood GP (variational/EP) per cell | included above | Calibration of variational posteriors needs its own check |
| MDO categorical kernel stalls | LVGP / descriptor kernel with screening P(wall) and derived features | 2 d | Descriptors are themselves noisy labels; propagate their Beta posteriors |
| Three seeds, HV only | >= 10 seeds, HV + IGD+ + epsilon, bootstrap CIs, Ishibuchi reference-point rule | +5-8x optimiser CPU (v2 was 83 min for 3 seeds) | Wall time under contention |
| Disjoint fronts under two closures | Report Ide-Schöbel sets (highly/flimsily robust, minmax regret); never pool | 1 d (analysis + dashboard) | None; it is the honest result |
| No physics-bearing closure | Calibrate p_k from PIC plateau wall fluxes at one point, re-check at a second (Marks-Jorns test) | 5-10 d + GPU time | Calibrated closure may not hold self-consistently |
| External validation 0 % | v0 code-to-code vs Greifswald PIC; v1 ion-energy peak / divergence angle vs Koch 2011 and HIT RPA in ASME V&V 20 form; v2 thrust-voltage vs mu-HEMPT/low-power HEMP; v3 geometry trends vs HIT | v0 5-8 d; v1 3-5 d after a refined-grid plateau; v2 10-20 d + data access | Published uncertainties missing (declare assumed u_D); device geometries may need author contact; our PIC device is not any published device |
| Preregistration lives only in Git; shakedowns see real data | External registry receipt; RO-Crate bundles; gate genealogy; synthetic/excluded-design shakedowns; ADEMP fields | 1-2 d once, then per campaign | None material |

## 6. Honest gaps

- No experimental dataset exists for the device family the repository actually simulates
  (2 mm bore, 300 V, mA-class). The closest published devices (mu-HEMPT, HIT low-power HEMP,
  DCFT) all require modelling *their* geometry; until that is done every comparison is
  same-class, not device validation.
- HEMP conference data (Thales/IEPC) publish few or no measurement uncertainties; ASME V&V 20
  cannot be applied without an assumed `u_D`, which must be declared as such.
- No TU Berlin HEMP dataset could be found; no "Ma 2024 AST" cusped-field MDO paper exists
  under that DOI (it is Yeo, Gadisa, Ogawa and Bang 2024 on FEEP).
- The published wall-loss-adjacent observables (thermal loss localisation, erosion, cusp
  leak width) are several inference steps from a collisionless test-particle probability;
  no literature comparison can make `GATE-WALL-LOSS-V4` or the screening dataset "validated".
- Most references were verified by resolving their DOI/arXiv/publisher record; only the
  IEPC-2007-108, IEPC-2011-236, Marks and Jorns (2023), Ma et al. (2015) abstract, Keller et
  al. (2015) abstract, Lazurenko et al. (2013) abstract and the MIT/HIT abstracts returned by
  the searches were read in text. Volume/page details are as returned by Crossref.
- The binomial-likelihood GP and the LVGP kernel are not in `cft_revival.surrogates` or the
  BoTorch adapter today; the recommendations assume they are added and tested first.
- The allocation figures in section 1.4 assume v1's throughput under the same CPU
  contention; a shakedown must re-measure it.

## 7. Bibliography count

157 references: 33 in section A, 35 in section B, 9 in section C, 16 in section D, 40 in section E, 24 in section F. 124 carry a DOI resolved through Crossref, 18 are arXiv-only, 15 are publisher/conference/standard URLs.

## 8. Bibliography

### A. Surrogates on stochastic (Monte-Carlo) labels, design of simulation experiments, model assessment (33)

- Ankenman B., Nelson B. L., Staum J. (2010). Stochastic Kriging for Simulation Metamodeling. *Operations Research 58(2):371-382*. https://doi.org/10.1287/opre.1090.0754
- Binois M., Gramacy R. B., Ludkovski M. (2018). Practical Heteroscedastic Gaussian Process Modeling for Large Simulation Experiments. *Journal of Computational and Graphical Statistics 27(4):808-821*. https://doi.org/10.1080/10618600.2018.1458625
- Binois M., Huang J., Gramacy R. B., Ludkovski M. (2019). Replication or Exploration? Sequential Design for Stochastic Simulation Experiments. *Technometrics 61(1):7-23*. https://doi.org/10.1080/00401706.2018.1469433
- Gramacy R. B. (2020). Surrogates. *Chapman and Hall/CRC*. https://doi.org/10.1201/9780367815493
- Gramacy R. B., Lee H. K. H. (2012). Cases for the nugget in modeling computer experiments. *Statistics and Computing 22(3):713-722*. https://doi.org/10.1007/s11222-010-9224-x
- Chen X., Ankenman B. E., Nelson B. L. (2012). The effects of common random numbers on stochastic kriging metamodels. *ACM Transactions on Modeling and Computer Simulation 22(2):1-20*. https://doi.org/10.1145/2133390.2133391
- Chen X., Zhou Q. (2017). Sequential design strategies for mean response surface metamodeling via stochastic kriging with adaptive exploration and exploitation. *European Journal of Operational Research 262(2):575-585*. https://doi.org/10.1016/j.ejor.2017.03.042
- Chen C. H., Lin J., Yücesan E., Chick S. E. (2000). Simulation Budget Allocation for Further Enhancing the Efficiency of Ordinal Optimization. *Discrete Event Dynamic Systems 10(3):251-270*. https://doi.org/10.1023/A:1008349927281
- Rasmussen C. E., Williams C. K. I. (2005). Gaussian Processes for Machine Learning. *The MIT Press*. https://doi.org/10.7551/mitpress/3206.001.0001 (MIT Press record year 2005; commonly cited as 2006)
- Williams C., Barber D. (1998). Bayesian classification with Gaussian processes. *IEEE Transactions on Pattern Analysis and Machine Intelligence 20(12):1342-1351*. https://doi.org/10.1109/34.735807
- Ferrari S., Cribari-Neto F. (2004). Beta Regression for Modelling Rates and Proportions. *Journal of Applied Statistics 31(7):799-815*. https://doi.org/10.1080/0266476042000214501
- Brown L. D., Cai T. T., DasGupta A. (2001). Interval Estimation for a Binomial Proportion. *Statistical Science 16(2)*. https://doi.org/10.1214/ss/1009213286
- Caflisch R. E. (1998). Monte Carlo and quasi-Monte Carlo methods. *Acta Numerica 7:1-49*. https://doi.org/10.1017/S0962492900002804
- Giles M. B. (2008). Multilevel Monte Carlo Path Simulation. *Operations Research 56(3):607-617*. https://doi.org/10.1287/opre.1070.0496
- Giles M. B. (2015). Multilevel Monte Carlo methods. *Acta Numerica 24:259-328*. https://doi.org/10.1017/S096249291500001X
- Kennedy M. C., O'Hagan A. (2000). Predicting the output from a complex computer code when fast approximations are available. *Biometrika 87(1):1-13*. https://doi.org/10.1093/biomet/87.1.1
- Peherstorfer B., Willcox K., Gunzburger M. (2018). Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization. *SIAM Review 60(3):550-591*. https://doi.org/10.1137/16M1082469
- Roustant O., Padonou E., Deville Y., Clément A., Perrin G., Giorla J. et al. (2020). Group Kernels for Gaussian Process Metamodels with Categorical Inputs. *SIAM/ASA Journal on Uncertainty Quantification 8(2):775-806*. https://doi.org/10.1137/18M1209386
- Zhang Y., Tao S., Chen W., Apley D. W. (2020). A Latent Variable Approach to Gaussian Process Modeling with Qualitative and Quantitative Factors. *Technometrics 62(3):291-302*. https://doi.org/10.1080/00401706.2019.1638834
- Pelamatti J., Brevault L., Balesdent M., Talbi E. G., Guerin Y. (2019). Efficient global optimization of constrained mixed variable problems. *Journal of Global Optimization 73(3):583-613*. https://doi.org/10.1007/s10898-018-0715-1
- Gramacy R. B., Lee H. K. H. (2008). Bayesian Treed Gaussian Process Models With an Application to Computer Modeling. *Journal of the American Statistical Association 103(483):1119-1130*. https://doi.org/10.1198/016214508000000689
- Sauer A., Gramacy R. B., Higdon D. (2023). Active Learning for Deep Gaussian Process Surrogates. *Technometrics 65(1):4-18*. https://doi.org/10.1080/00401706.2021.2008505
- Nadeau C., Bengio Y. (2003). Inference for the Generalization Error. *Machine Learning 52(3):239-281*. https://doi.org/10.1023/A:1024068626366
- Varma S., Simon R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics 7(1)*. https://doi.org/10.1186/1471-2105-7-91
- Viering T., Loog M. (2023). The Shape of Learning Curves: A Review. *IEEE Transactions on Pattern Analysis and Machine Intelligence 45(6):7799-7819*. https://doi.org/10.1109/TPAMI.2022.3220744
- Kapoor S., Narayanan A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns 4(9):100804*. https://doi.org/10.1016/j.patter.2023.100804
- Saltelli A., Annoni P., Azzini I., Campolongo F., Ratto M., Tarantola S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications 181(2):259-270*. https://doi.org/10.1016/j.cpc.2009.09.018
- Hensman J., Matthews A. G. de G., Ghahramani Z. (2015). Scalable Variational Gaussian Process Classification. *Proc. 18th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 38*. arXiv:1411.2005. https://arxiv.org/abs/1411.2005
- Kuss M., Rasmussen C. E. (2005). Assessing Approximate Inference for Binary Gaussian Process Classification. *Journal of Machine Learning Research 6:1679-1704*. https://www.jmlr.org/papers/v6/kuss05a.html
- Damianou A. C., Lawrence N. D. (2013). Deep Gaussian Processes. *Proc. 16th AISTATS, PMLR 31*. arXiv:1211.0358. https://arxiv.org/abs/1211.0358
- Cawley G. C., Talbot N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine Learning Research 11:2079-2107*. https://www.jmlr.org/papers/v11/cawley10a.html
- Demšar J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research 7:1-30*. https://www.jmlr.org/papers/v7/demsar06a.html
- Hestness J., Narang S., Ardalani N., Diamos G., Jun H., Kianinejad H. et al. (2017). Deep Learning Scaling is Predictable, Empirically. arXiv:1712.00409. https://arxiv.org/abs/1712.00409

### B. Bayesian and robust multi-objective optimisation, model-form uncertainty, benchmarking (35)

- Kennedy M. C., O'Hagan A. (2001). Bayesian Calibration of Computer Models. *Journal of the Royal Statistical Society Series B: Statistical Methodology 63(3):425-464*. https://doi.org/10.1111/1467-9868.00294
- Brynjarsdóttir J., O'Hagan A. (2014). Learning about physical parameters: the importance of model discrepancy. *Inverse Problems 30(11):114007*. https://doi.org/10.1088/0266-5611/30/11/114007
- Roy C. J., Oberkampf W. L. (2011). A comprehensive framework for verification, validation, and uncertainty quantification in scientific computing. *Computer Methods in Applied Mechanics and Engineering 200(25-28):2131-2144*. https://doi.org/10.1016/j.cma.2011.03.016
- Riley M. E., Grandhi R. V. (2011). Quantification of model-form and predictive uncertainty for multi-physics simulation. *Computers & Structures 89(11-12):1206-1213*. https://doi.org/10.1016/j.compstruc.2010.10.004
- Tebaldi C., Knutti R. (2007). The use of the multi-model ensemble in probabilistic climate projections. *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 365(1857):2053-2075*. https://doi.org/10.1098/rsta.2007.2076
- Beyer H. G., Sendhoff B. (2007). Robust optimization – A comprehensive survey. *Computer Methods in Applied Mechanics and Engineering 196(33-34):3190-3218*. https://doi.org/10.1016/j.cma.2007.03.003
- Deb K., Gupta H. (2006). Introducing Robustness in Multi-Objective Optimization. *Evolutionary Computation 14(4):463-494*. https://doi.org/10.1162/evco.2006.14.4.463
- Ide J., Schöbel A. (2016). Robustness for uncertain multi-objective optimization: a survey and analysis of different concepts. *OR Spectrum 38(1):235-271*. https://doi.org/10.1007/s00291-015-0418-7
- Zitzler E., Thiele L., Laumanns M., Fonseca C., da Fonseca V. (2003). Performance assessment of multiobjective optimizers: an analysis and review. *IEEE Transactions on Evolutionary Computation 7(2):117-132*. https://doi.org/10.1109/TEVC.2003.810758
- Ishibuchi H., Imada R., Setoguchi Y., Nojima Y. (2018). How to Specify a Reference Point in Hypervolume Calculation for Fair Performance Comparison. *Evolutionary Computation 26(3):411-440*. https://doi.org/10.1162/evco_a_00226
- Audet C., Bigeon J., Cartier D., Le Digabel S., Salomon L. (2021). Performance indicators in multiobjective optimization. *European Journal of Operational Research 292(2):397-422*. https://doi.org/10.1016/j.ejor.2020.11.016
- Derrac J., García S., Molina D., Herrera F. (2011). A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms. *Swarm and Evolutionary Computation 1(1):3-18*. https://doi.org/10.1016/j.swevo.2011.02.002
- Hansen N., Auger A., Ros R., Mersmann O., Tušar T., Brockhoff D. (2021). COCO: a platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software 36(1):114-144*. https://doi.org/10.1080/10556788.2020.1808977
- Martins J. R. R. A., Lambe A. B. (2013). Multidisciplinary Design Optimization: A Survey of Architectures. *AIAA Journal 51(9):2049-2075*. https://doi.org/10.2514/1.J051895
- Yao W., Chen X., Luo W., van Tooren M., Guo J. (2011). Review of uncertainty-based multidisciplinary design optimization methods for aerospace vehicles. *Progress in Aerospace Sciences 47(6):450-479*. https://doi.org/10.1016/j.paerosci.2011.05.001
- Jorns B. (2018). Predictive, data-driven model for the anomalous electron collision frequency in a Hall effect thruster. *Plasma Sources Science and Technology 27(10):104007*. https://doi.org/10.1088/1361-6595/aae472
- Marks T. A., Jorns B. A. (2023). Challenges with the self-consistent implementation of closure models for anomalous electron transport in fluid simulations of Hall thrusters. *Plasma Sources Science and Technology 32(4):045016*. https://doi.org/10.1088/1361-6595/accd18 (abstract read)
- Mikellides I. G., Lopez Ortega A. (2019). Challenges in the development and verification of first-principles models in Hall-effect thruster simulations that are based on anomalous resistivity and generalized Ohm's law. *Plasma Sources Science and Technology 28(1):014003*. https://doi.org/10.1088/1361-6595/aae63b
- Hara K. (2019). An overview of discharge plasma modeling for Hall effect thrusters. *Plasma Sources Science and Technology 28(4):044001*. https://doi.org/10.1088/1361-6595/ab0f70
- Boeuf J. P. (2017). Tutorial: Physics and modeling of Hall thrusters. *Journal of Applied Physics 121(1)*. https://doi.org/10.1063/1.4972269
- Kaganovich I. D., Smolyakov A., Raitses Y., Ahedo E., Mikellides I. G., Jorns B. et al. (2020). Physics of E x B discharges relevant to plasma propulsion and similar technologies. *Physics of Plasmas 27(12)*. https://doi.org/10.1063/5.0010135
- Balandat M., Karrer B., Jiang D. R., Daulton S., Letham B., Wilson A. G., Bakshy E. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. *Advances in Neural Information Processing Systems 33*. arXiv:1910.06403. https://arxiv.org/abs/1910.06403
- Daulton S., Balandat M., Bakshy E. (2020). Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. *Advances in Neural Information Processing Systems 33*. arXiv:2006.05078. https://arxiv.org/abs/2006.05078
- Daulton S., Balandat M., Bakshy E. (2021). Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. *Advances in Neural Information Processing Systems 34*. arXiv:2105.08195. https://arxiv.org/abs/2105.08195
- Ament S., Daulton S., Eriksson D., Balandat M., Bakshy E. (2023). Unexpected Improvements to Expected Improvement for Bayesian Optimization. *Advances in Neural Information Processing Systems 36*. arXiv:2310.20708. https://arxiv.org/abs/2310.20708
- Daulton S., Cakmak S., Balandat M., Osborne M. A., Zhou E., Bakshy E. (2022). Robust Multi-Objective Bayesian Optimization Under Input Noise. *Proc. 39th International Conference on Machine Learning (ICML), PMLR 162*. arXiv:2202.07549. https://arxiv.org/abs/2202.07549
- Cakmak S., Astudillo R., Frazier P., Zhou E. (2020). Bayesian Optimization of Risk Measures. *Advances in Neural Information Processing Systems 33*. arXiv:2007.05554. https://arxiv.org/abs/2007.05554
- Eriksson D., Poloczek M. (2021). Scalable Constrained Bayesian Optimization. *Proc. 24th AISTATS, PMLR 130*. arXiv:2002.08526. https://arxiv.org/abs/2002.08526
- Daulton S., Eriksson D., Balandat M., Bakshy E. (2022). Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces. *Proc. 38th Conference on Uncertainty in Artificial Intelligence (UAI), PMLR 180*. arXiv:2109.10964. https://arxiv.org/abs/2109.10964
- Daulton S., Wan X., Eriksson D., Balandat M., Osborne M. A., Bakshy E. (2022). Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization. *Advances in Neural Information Processing Systems 35*. arXiv:2210.10199. https://arxiv.org/abs/2210.10199
- Ru B., Alvi A. S., Nguyen V., Osborne M. A., Roberts S. (2020). Bayesian Optimisation over Multiple Continuous and Categorical Inputs. *Proc. 37th ICML, PMLR 119*. arXiv:1906.08878. https://arxiv.org/abs/1906.08878
- Wan X., Nguyen V., Ha H., Ru B., Lu C., Osborne M. A. (2021). Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. *Proc. 38th ICML, PMLR 139*. arXiv:2102.07188. https://arxiv.org/abs/2102.07188
- Frazier P. I. (2018). A Tutorial on Bayesian Optimization. arXiv:1807.02811. https://arxiv.org/abs/1807.02811
- Bartz-Beielstein T., Doerr C., van den Berg D., Bossek J., Chandrasekaran S., Eftimov T. et al. (2020). Benchmarking in Optimization: Best Practice and Open Issues. arXiv:2007.03488. https://arxiv.org/abs/2007.03488
- Henderson P., Islam R., Bachman P., Pineau J., Precup D., Meger D. (2018). Deep Reinforcement Learning that Matters. *Proc. 32nd AAAI Conference on Artificial Intelligence*. arXiv:1709.06560. https://arxiv.org/abs/1709.06560

### C. Electric-propulsion design-optimisation lineage (9)

- Fahey T., Muffatti A., Ogawa H. (2017). High Fidelity Multi-Objective Design Optimization of a Downscaled Cusped Field Thruster. *Aerospace 4(4):55*. https://doi.org/10.3390/aerospace4040055 (methods section read)
- Yeo S. H., Ogawa H. (2019). Investigation of Influence of Magnet Thickness on Performance of Cusped Field Thruster via Multi-objective Design Optimization. *Lecture Notes in Electrical Engineering:1969-1989*. https://doi.org/10.1007/978-981-13-3305-7_159
- Yeo S. H., Ogawa H., Matthias P., Kahnfeld D., Schneider R. (2020). Multiobjective Optimization and Particle-In-Cell Simulation of Cusped Field Thrusters for Microsatellite Platforms. *Journal of Spacecraft and Rockets 57(3):603-611*. https://doi.org/10.2514/1.A34584 (abstract read; S1 values also in `modern/data/validation/`)
- Yeo S. H., Ogawa H. (2022). Multi-Objective Design Optimization of Cusped Field Thruster via Surrogate-Assisted Evolutionary Algorithms. *Journal of Propulsion and Power 38(6):973-988*. https://doi.org/10.2514/1.B38854
- Yeo S. H., Ogawa H. (2023). Multi-objective Design Optimization and Uncertainty Analysis of a Downscaled Cusped Field Thruster. *Lecture Notes in Electrical Engineering:1397-1410*. https://doi.org/10.1007/978-981-19-2635-8_99
- Yeo S. H., Gadisa D., Ogawa H., Bang H. (2024). Multi-objective design optimization and physics-based sensitivity analysis of field emission electric propulsion for CubeSat platforms. *Aerospace Science and Technology 154:109516*. https://doi.org/10.1016/j.ast.2024.109516
- Lewerentz L., Schneider R. (2023). Simplified Optimization of the Magnetic Configuration of HEMP-Thrusters. *Applied Sciences 13(6):3491*. https://doi.org/10.3390/app13063491
- Puca N., Panelli M., Battista F. (2024). A Methodology for the Preliminary Design of a High-Efficiency Multistage Plasma Thruster. *Aerotecnica Missili & Spazio 103(4):321-338*. https://doi.org/10.1007/s42496-024-00203-x (abstract read)
- Muffatti A., Ogawa H. (2017). Multi-objective Design Optimisation of a Small Scale Cusped Field Thruster for Micro-satellite Platforms. *31st International Symposium on Space Technology and Science, ISTS 2017-b-32*. https://archive.ists.ne.jp/upload_pdf/2017-b-32.pdf (URL opened; SHA-256 recorded in `modern/docs/REFERENCES.md`)

### D. Verification, validation and uncertainty quantification frameworks (16)

- Oberkampf W. L., Roy C. J. (2010). Verification and Validation in Scientific Computing. *Cambridge University Press*. https://doi.org/10.1017/CBO9780511760396
- Oberkampf W. L., Trucano T. G. (2002). Verification and validation in computational fluid dynamics. *Progress in Aerospace Sciences 38(3):209-272*. https://doi.org/10.1016/S0376-0421(02)00005-2
- Oberkampf W. L., Barone M. F. (2006). Measures of agreement between computation and experiment: Validation metrics. *Journal of Computational Physics 217(1):5-36*. https://doi.org/10.1016/j.jcp.2006.03.037
- Ferson S., Oberkampf W. L., Ginzburg L. (2008). Model validation and predictive capability for the thermal challenge problem. *Computer Methods in Applied Mechanics and Engineering 197(29-32):2408-2430*. https://doi.org/10.1016/j.cma.2007.07.030
- Liu Y., Chen W., Arendt P., Huang H. Z. (2011). Toward a Better Understanding of Model Validation Metrics. *Journal of Mechanical Design 133(7)*. https://doi.org/10.1115/1.4004223
- Bayarri M. J., Berger J. O., Paulo R., Sacks J., Cafeo J. A., Cavendish J. et al. (2007). A Framework for Validation of Computer Models. *Technometrics 49(2):138-154*. https://doi.org/10.1198/004017007000000092
- Trucano T., Swiler L., Igusa T., Oberkampf W., Pilch M. (2006). Calibration, validation, and sensitivity analysis: What's what. *Reliability Engineering & System Safety 91(10-11):1331-1357*. https://doi.org/10.1016/j.ress.2005.11.031
- National Research Council (2012). Assessing the Reliability of Complex Models. *National Academies Press*. https://doi.org/10.17226/13395
- Oberkampf W., Trucano T., Pilch M. (2007). Predictive Capability Maturity Model for computational modeling and simulation (SAND2007-5948). *Office of Scientific and Technical Information (OSTI)*. https://doi.org/10.2172/976951
- Greenwald M. (2010). Verification and validation for magnetic fusion. *Physics of Plasmas 17(5)*. https://doi.org/10.1063/1.3298884
- Terry P. W., Greenwald M., Leboeuf J. N., McKee G. R., Mikkelsen D. R., Nevins W. M. et al. (2008). Validation in fusion research: Towards guidelines and best practices. *Physics of Plasmas 15(6)*. https://doi.org/10.1063/1.2928909
- Holland C. (2016). Validation metrics for turbulent plasma transport. *Physics of Plasmas 23(6)*. https://doi.org/10.1063/1.4954151
- Charoy T., Boeuf J. P., Bourdon A., Carlsson J. A., Chabert P., Cuenot B. et al. (2019). 2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized plasmas. *Plasma Sources Science and Technology 28(10):105010*. https://doi.org/10.1088/1361-6595/ab46c5
- Villafana W., Petronio F., Denig A. C., Jimenez M. J., Eremin D., Garrigues L. et al. (2021). 2D radial-azimuthal particle-in-cell benchmark for E x B discharges. *Plasma Sources Science and Technology 30(7):075002*. https://doi.org/10.1088/1361-6595/ac0a4a
- ASME (2009). *V&V 20-2009: Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer* (reaffirmed 2016 and 2021). The American Society of Mechanical Engineers, New York. https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer
- NASA (2016). *NASA-STD-7009A: Standard for Models and Simulations* (Revision A, 2016-07-13). National Aeronautics and Space Administration. https://ntrs.nasa.gov/api/citations/20160011121/downloads/20160011121.pdf

### E. HEMP / cusped-field thruster experiments and simulations (40)

- Koch N., Weis S., Schirra M., Lazurenko A., van Reijen B., Haderspeck J. et al. (2011). The High Efficiency Multistage Plasma Thruster HEMPT based Ion Propulsion System for the SmallGEO Satellite. *47th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &amp; Exhibit*. https://doi.org/10.2514/6.2011-6086
- Matyash K., Schneider R., Mutzke A., Kalentev O., Taccogna F., Koch N. et al. (2010). Kinetic Simulations of SPT and HEMP Thrusters Including the Near-Field Plume Region. *IEEE Transactions on Plasma Science 38(9):2274-2280*. https://doi.org/10.1109/TPS.2010.2056936 (also arXiv:0912.0470)
- Duras J., Kahnfeld D., Bandelow G., Kemnitz S., Lüskow K., Matthias P. et al. (2017). Ion angular distribution simulation of the Highly Efficient Multistage Plasma Thruster. *Journal of Plasma Physics 83(1)*. https://doi.org/10.1017/S0022377817000125 (abstract read)
- Duras J., Kalentev O., Schneider R., Matyash K., Lüskow K. F., Geiser J. (2015). Electrostatic ion thrusters - towards predictive modeling. *Acta Polytechnica 55(1):7-13*. https://doi.org/10.14311/ap.2015.55.0007
- Kahnfeld D., Duras J., Matthias P., Kemnitz S., Arlinghaus P., Bandelow G. et al. (2019). Numerical modeling of high efficiency multistage plasma thrusters for space applications. *Reviews of Modern Plasma Physics 3(1)*. https://doi.org/10.1007/s41614-019-0030-4
- Matthias P., Kahnfeld D., Schneider R., Yeo S. H., Ogawa H. (2019). Particle-in-cell simulation of an optimized high-efficiency multistage plasma thruster. *Contributions to Plasma Physics 59(9)*. https://doi.org/10.1002/ctpp.201900028
- Lewerentz L., Kahnfeld D., Schulz N., Heidemann R., Schneider R. (2022). PIC Simulations of the MS4 Thruster. *Frontiers in Physics 10*. https://doi.org/10.3389/fphy.2022.833159
- Keller A., Kohler P., Hey F. G., Berger M., Braxmaier C., Feili D. et al. (2015). Parametric Study of HEMP-Thruster Downscaling to µN Thrust Levels. *IEEE Transactions on Plasma Science 43(1):45-53*. https://doi.org/10.1109/TPS.2014.2321095 (abstract read)
- Hey F. G., Keller A., Braxmaier C., Tajmar M., Johann U., Weise D. (2015). Development of a Highly Precise Micronewton Thrust Balance. *IEEE Transactions on Plasma Science 43(1):234-239*. https://doi.org/10.1109/tps.2014.2377652 (abstract read)
- Hey F. G., Keller A., Johann U., Braxmaier C., Tajmar M., Fitzsimons E. et al. (2015). Development of a Micro-Thruster Test Facility which fulfils the LISA requirements. *Journal of Physics: Conference Series 610:012037*. https://doi.org/10.1088/1742-6596/610/1/012037
- Neumann A. (2017). Update on Diagnostics for DLR's Electric Propulsion Test Facility. *Procedia Engineering 185:47-52*. https://doi.org/10.1016/j.proeng.2017.03.289
- Ma C., Liu H., Hu Y., Yu D., Chen P., Sun G. et al. (2015). Experimental study on a variable magnet length cusped field thruster. *Vacuum 115:101-107*. https://doi.org/10.1016/j.vacuum.2015.02.007 (abstract read)
- Hu P., Liu H., Gao Y., Mao W., Yu D. (2016). An experimental study of the effect of magnet length on the performance of a multi-cusped field thruster. *Journal of Physics D: Applied Physics 49(28):285201*. https://doi.org/10.1088/0022-3727/49/28/285201
- Hu P., Liu H., Gao Y., Yu D. (2016). Effects of magnetic field strength in the discharge channel on the performance of a multi-cusped field thruster. *AIP Advances 6(9)*. https://doi.org/10.1063/1.4962548 (abstract read)
- Hu P., Liu H., Mao W., Yu D., Gao Y. (2015). The effects of magnetic field in plume region on the performance of multi-cusped field thruster. *Physics of Plasmas 22(10)*. https://doi.org/10.1063/1.4932077 (abstract read)
- Hu P., Liu H., Gao Y., Yu D. (2016). Study on the structure and transition of the hollow plume in a multi-cusped field thruster. *Physics of Plasmas 23(10)*. https://doi.org/10.1063/1.4965910 (abstract read)
- Wu H., Liu H., Meng Y., Zhang J., Yang S., Hu P. et al. (2015). Anode current density distribution in a cusped field thruster. *Physics of Plasmas 22(12)*. https://doi.org/10.1063/1.4938042 (abstract read)
- Liu H., Wu H., Zhao Y., Yu D., Ma C., Wang D. et al. (2014). Study of the electric field formation in a multi-cusped magnetic field. *Physics of Plasmas 21(9)*. https://doi.org/10.1063/1.4896250 (abstract read)
- Zhao Y. J., Liu H., Yu D. R., Hu P., Wu H. (2014). Particle-in-cell simulations for the effect of magnetic field strength on a cusped field thruster. *Journal of Physics D: Applied Physics 47(4):045201*. https://doi.org/10.1088/0022-3727/47/4/045201
- Liu H., Zeng M., Yu D., Huang H. (2019). Study of channel length effect on low power HEMP Thruster. *Vacuum 163:328-337*. https://doi.org/10.1016/j.vacuum.2019.02.035
- Courtney D., Lozano P., Martinez-Sanchez M. (2008). Continued Investigation of Diverging Cusped Field Thruster. *44th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &amp; Exhibit*. https://doi.org/10.2514/6.2008-4631
- Gildea S., Batishchev O., Martinez-Sanchez M. (2009). Fully Kinetic Modeling of Divergent Cusped Field Thrusters. *45th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &amp; Exhibit*. https://doi.org/10.2514/6.2009-4814
- Matlock T., Gildea S., Hu F., Becker N., Lozano P., Martinez-Sanchez M. (2010). Magnetic Field Effects on the Plume of a Diverging Cusped-Field Thruster. *46th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &amp; Exhibit*. https://doi.org/10.2514/6.2010-7104 (abstract read)
- Gildea S., Matlock T., Lozano P., Martinez-Sanchez M. (2010). Low Frequency Oscillations in the Diverging Cusped-Field Thruster. *46th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &amp; Exhibit*. https://doi.org/10.2514/6.2010-7014
- Daspit R., Lozano P., Martinez-Sanchez M. (2011). Characterization and Optimization of a Diverging Cusped Field Thruster with a Calibrated Horizontal Accelerometer. *47th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &amp; Exhibit*. https://doi.org/10.2514/6.2011-6069
- MacDonald N. A., Cappelli M. A., Gildea S. R., Martínez-Sánchez M., Hargus W. A. (2011). Laser-induced fluorescence velocity measurements of a diverging cusped-field thruster. *Journal of Physics D: Applied Physics 44(29):295203*. https://doi.org/10.1088/0022-3727/44/29/295203 (abstract read)
- MacDonald N. A., Young C. V., Cappelli M. A., Hargus W. A. (2012). Ion velocity and plasma potential measurements of a cylindrical cusped field thruster. *Journal of Applied Physics 111(9)*. https://doi.org/10.1063/1.4707953
- Gildea S. R., Matlock T. S., Martínez-Sánchez M., Hargus W. A. (2013). Erosion Measurements in a Low-Power Cusped-Field Plasma Thruster. *Journal of Propulsion and Power 29(4):906-918*. https://doi.org/10.2514/1.B34607
- Raitses Y., Fisch N. J. (2001). Parametric investigations of a nonconventional Hall thruster. *Physics of Plasmas 8(5):2579-2586*. https://doi.org/10.1063/1.1355318
- Smirnov A., Raitses Y., Fisch N. J. (2002). Parametric investigation of miniaturized cylindrical and annular Hall thrusters. *Journal of Applied Physics 92(10):5673-5679*. https://doi.org/10.1063/1.1515106
- Smirnov A., Raitses Y., Fisch N. J. (2004). Plasma measurements in a 100 W cylindrical Hall thruster. *Journal of Applied Physics 95(5):2283-2292*. https://doi.org/10.1063/1.1642734
- Raitses Y., Smirnov A., Fisch N. J. (2007). Enhanced performance of cylindrical Hall thrusters. *Applied Physics Letters 90(22)*. https://doi.org/10.1063/1.2741413
- Leung K. N., Hershkowitz N., MacKenzie K. R. (1976). Plasma confinement by localized cusps. *The Physics of Fluids 19(7):1045-1053*. https://doi.org/10.1063/1.861575
- Kornfeld G., Koch N., Harmann H.-P. (2007). Physics and Evolution of HEMP-Thrusters. *30th International Electric Propulsion Conference, Florence, IEPC-2007-108*. https://electricrocket.org/IEPC/IEPC-2007-108.pdf (full text read)
- Koch N., Harmann H.-P., Kornfeld G. (2007). Status of the THALES High Efficiency Multi Stage Plasma Thruster Development for HEMP-T 3050 and HEMP-T 30250. *30th International Electric Propulsion Conference, Florence, IEPC-2007-110*. https://electricrocket.org/IEPC/IEPC-2007-110.pdf
- Koch N., Schirra M., Weis S., Lazurenko A., van Reijen B., Haderspeck J., Genovese A., Holtmann P., Schneider R., Matyash K., Kalentyev O. (2011). The HEMPT Concept - A Survey on Theoretical Considerations and Experimental Evidences. *32nd International Electric Propulsion Conference, Wiesbaden, IEPC-2011-236*. https://electricrocket.org/IEPC/IEPC-2011-236.pdf (full text read)
- Genovese A., Lazurenko A., Koch N., Weis S., Schirra M., van Reijen B., Haderspeck J., Holtmann P. (2011). Endurance Testing of HEMPT-based Ion Propulsion Modules for SmallGEO. *32nd International Electric Propulsion Conference, Wiesbaden, IEPC-2011-141*. https://electricrocket.org/IEPC/IEPC-2011-141.pdf
- Keller A., Köhler P., Feili D., Berger M., Braxmaier C., Weise D., Johann U. (2011). Feasibility of a down-scaled HEMP-Thruster. *32nd International Electric Propulsion Conference, Wiesbaden, IEPC-2011-138*. https://electricrocket.org/IEPC/IEPC-2011-138.pdf
- Lazurenko A., Genovese A., van Reijen B., Haderspeck J., Schirra M., Weis S. et al. (2013). Progress in Testing of QM and FM HEMP Thruster Modules. *33rd International Electric Propulsion Conference, Washington DC, IEPC-2013-339*. http://erps.spacegrant.org/uploads/images/images/iepc_articledownload_1988-2007/2013index/ls1bb3r6.pdf (index-verified; direct fetch refused at review time)
- Gildea S. R., Martinez-Sanchez M., Nakles M. R., Hargus W. A. (2009). Experimentally Characterizing the Plume of a Divergent Cusped-Field Thruster. *31st International Electric Propulsion Conference, Ann Arbor, IEPC-2009-259*. https://electricrocket.org/IEPC/IEPC-2009-259.pdf

### F. Reproducibility, preregistration and research-artefact practice (24)

- Nosek B. A., Ebersole C. R., DeHaven A. C., Mellor D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences 115(11):2600-2606*. https://doi.org/10.1073/pnas.1708274114
- Chambers C. D., Tzavella L. (2021). The past, present and future of Registered Reports. *Nature Human Behaviour 6(1):29-42*. https://doi.org/10.1038/s41562-021-01193-7 (Crossref record year 2021; print volume 6 is 2022)
- Hardwicke T. E., Wagenmakers E. J. (2023). Reducing bias, increasing transparency and calibrating confidence with preregistration. *Nature Human Behaviour 7(1):15-26*. https://doi.org/10.1038/s41562-022-01497-2
- Gelman A., Loken E. (2014). The Statistical Crisis in Science. *American Scientist 102(6):460*. https://doi.org/10.1511/2014.111.460
- Simmons J. P., Nelson L. D., Simonsohn U. (2011). False-Positive Psychology. *Psychological Science 22(11):1359-1366*. https://doi.org/10.1177/0956797611417632
- Sagarin B. J., Ambler J. K., Lee E. M. (2014). An Ethical Approach to Peeking at Data. *Perspectives on Psychological Science 9(3):293-304*. https://doi.org/10.1177/1745691614528214
- Albers C., Lakens D. (2018). When power analyses based on pilot data are biased: Inaccurate effect size estimators and follow-up bias. *Journal of Experimental Social Psychology 74:187-195*. https://doi.org/10.1016/j.jesp.2017.09.004
- Crüwell S., Evans N. J. (2021). Preregistration in diverse contexts: a preregistration template for the application of cognitive models. *Royal Society Open Science 8(10)*. https://doi.org/10.1098/rsos.210155
- Morris T. P., White I. R., Crowther M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine 38(11):2074-2102*. https://doi.org/10.1002/sim.8086
- Siepe B. S., Bartoš F., Morris T. P., Boulesteix A. L., Heck D. W., Pawel S. (2024). Simulation studies for methodological research in psychology: A standardized template for planning, preregistration, and reporting. *Psychological Methods*. https://doi.org/10.1037/met0000695
- Pawel S., Kook L., Reeve K. (2024). Pitfalls and potentials in simulation studies: Questionable research practices in comparative simulation studies allow for spurious claims of superiority of any method. *Biometrical Journal 66(1)*. https://doi.org/10.1002/bimj.202200091
- Boulesteix A. L., Lauer S., Eugster M. J. A. (2013). A Plea for Neutral Comparison Studies in Computational Sciences. *PLoS ONE 8(4):e61562*. https://doi.org/10.1371/journal.pone.0061562
- Nießl C., Herrmann M., Wiedemann C., Casalicchio G., Boulesteix A. (2022). Over-optimism in benchmark studies and the multiplicity of design and analysis options when interpreting their results. *WIREs Data Mining and Knowledge Discovery 12(2)*. https://doi.org/10.1002/widm.1441
- Cockburn A., Dragicevic P., Besançon L., Gutwin C. (2020). Threats of a replication crisis in empirical computer science. *Communications of the ACM 63(8):70-79*. https://doi.org/10.1145/3360311
- Heil B. J., Hoffman M. M., Markowetz F., Lee S. I., Greene C. S., Hicks S. C. (2021). Reproducibility standards for machine learning in the life sciences. *Nature Methods 18(10):1132-1135*. https://doi.org/10.1038/s41592-021-01256-7
- Wilkinson M. D., Dumontier M., Aalbersberg I. J., Appleton G., Axton M., Baak A. et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data 3(1)*. https://doi.org/10.1038/sdata.2016.18
- Soiland-Reyes S., Sefton P., Crosas M., Castro L. J., Coppens F., Fernández J. M. et al. (2022). Packaging research artefacts with RO-Crate. *Data Science 5(2):97-138*. https://doi.org/10.3233/DS-210053
- Peng R. D. (2011). Reproducible Research in Computational Science. *Science 334(6060):1226-1227*. https://doi.org/10.1126/science.1213847
- Stodden V., McNutt M., Bailey D. H., Deelman E., Gil Y., Hanson B. et al. (2016). Enhancing reproducibility for computational methods. *Science 354(6317):1240-1241*. https://doi.org/10.1126/science.aah6168
- Sandve G. K., Nekrutenko A., Taylor J., Hovig E. (2013). Ten Simple Rules for Reproducible Computational Research. *PLoS Computational Biology 9(10):e1003285*. https://doi.org/10.1371/journal.pcbi.1003285
- Ivie P., Thain D. (2019). Reproducibility in Scientific Computing. *ACM Computing Surveys 51(3):1-36*. https://doi.org/10.1145/3186266
- Bertinetto L., Henriques J. F., Albanie S., Paganini M., Varol G. (eds) (2021). NeurIPS 2020 Workshop on Pre-registration in Machine Learning. *Proceedings of Machine Learning Research 148*. https://proceedings.mlr.press/v148/
- Pineau J., Vincent-Lamarre P., Sinha K., Larivière V., Beygelzimer A., d'Alché-Buc F., Fox E., Larochelle H. (2021). Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research 22(164):1-20*. https://www.jmlr.org/papers/v22/20-303.html
- Barba L. A. (2018). Terminologies for Reproducible Research. arXiv:1802.03311. https://arxiv.org/abs/1802.03311

